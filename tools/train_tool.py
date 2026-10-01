import json
import logging
import shutil
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from tools.eval_tool import gen_time_str, output_value, valid
from utils.device import (
    get_device,
    prepare_data_loader,
    set_data_loader_epoch,
    validate_xla_batch_shape,
    xm,
)
from utils.paths import PathManager

logger = logging.getLogger(__name__)


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def _is_primary_process(device):
    return device.type != "xla" or xm is None or xm.is_master_ordinal(local=False)


def _distributed_validation_loss(value, device, epoch):
    if device.type != "xla" or xm is None:
        return value
    return xm.mesh_reduce(
        f"validation_loss_{epoch}",
        value,
        lambda values: sum(values) / len(values),
    )


def _configure_mixed_precision(device, precision):
    if device.type == "xla" and precision == "fp16":
        raise ValueError("TPU/XLA não suporta fp16 neste pipeline; use precision=bf16 ou fp32.")

    use_amp = precision in ("fp16", "bf16") and device.type in ("cuda", "cpu", "xla")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    scaler = torch.amp.GradScaler(
        scaler_device,
        enabled=use_amp and precision == "fp16" and device.type != "xla",
    )
    return use_amp, amp_dtype, scaler


def _optimizer_step(optimizer, scaler, device):
    if device.type == "xla":
        if xm is None:
            raise RuntimeError("Dispositivo XLA selecionado, mas torch_xla não está disponível.")
        xm.optimizer_step(optimizer, barrier=True)
        return

    scaler.step(optimizer)
    scaler.update()


def _save_checkpoint(save_params, filename, device=None):
    if device is not None and device.type == "xla":
        if xm is None:
            raise RuntimeError("Dispositivo XLA selecionado, mas torch_xla não está disponível.")
        xm.mark_step()
        xm.wait_device_ops()
        xm.save(save_params, filename, master_only=True)
        return

    torch.save(save_params, filename)


def checkpoint(
    filename,
    model,
    optimizer,
    trained_epoch,
    config,
    global_step,
    warmup_scheduler=None,
    device=None,
):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
    if warmup_scheduler is not None:
        save_params["warmup_scheduler"] = warmup_scheduler.state_dict()

    try:
        _save_checkpoint(save_params, filename, device)
    except Exception as e:
        logger.warning(f"Cannot save models with error {e!s}, continue anyway")


def train(parameters, config, gpu_list):
    epoch = config.getint("train", "epoch")
    # batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = Path(config.get("output", "model_path")) / config.get("output", "model_name")
    if output_path.exists():
        logger.warning("Output path exists, check whether need to change a name of model")
    PathManager.ensure_dir(output_path)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    _initial_global_step = global_step
    output_function = parameters["output_function"]

    # ── Device portável (CUDA / MPS / CPU) ──────────────────────────────────
    device = get_device()
    dataset = prepare_data_loader(dataset, device)

    # ── Gradient clipping (padrão recomendado para fine-tuning de BERT) ──────
    max_grad_norm = config.getfloat("train", "max_grad_norm", fallback=1.0)

    # ── Mixed Precision (AMP) ────────────────────────────────────────────────
    precision = config.get("environment", "precision", fallback="fp32")
    use_amp, amp_dtype, scaler = _configure_mixed_precision(device, precision)
    if use_amp:
        logger.info("AMP habilitado: dtype=%s, GradScaler=%s", amp_dtype, scaler.is_enabled())

    # Profiling metrics storage
    profiling_metrics = {
        "total_flops": 0,
        "avg_flops_per_batch": 0,
        "profiled_batches": 0
    }

    tensorboard_path = Path(config.get("output", "tensorboard_path")) / config.get("output", "model_name")
    
    if trained_epoch == 0 and _is_primary_process(device):
        shutil.rmtree(tensorboard_path, ignore_errors=True)

    PathManager.ensure_dir(tensorboard_path)

    writer = (
        SummaryWriter(str(tensorboard_path), config.get("output", "model_name"))
        if _is_primary_process(device)
        else _NullSummaryWriter()
    )

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Scheduler linear com warmup para bert_adam (substitui o StepLR nesse caso)
    warmup_scheduler = None
    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "bert_adam":
        total_steps = len(dataset) * (epoch - (parameters["trained_epoch"] + 1))
        num_warmup_steps = int(
            config.getfloat("train", "warmup_ratio", fallback=0.1) * total_steps
        )
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        if "warmup_scheduler_state" in parameters:
            warmup_scheduler.load_state_dict(parameters["warmup_scheduler_state"])
            logger.info("Warmup scheduler state restored from checkpoint.")
        logger.info(
            "Warmup scheduler criado: total_steps=%d, warmup_steps=%d",
            total_steps, num_warmup_steps
        )

    logger.info("Training start....")

    # ── Early Stopping ───────────────────────────────────────────────────────────
    es_patience = config.getint("train", "early_stopping_patience", fallback=0)
    es_counter = 0
    best_val_loss = float("inf")
    early_stopped = False

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    if total_len < 10000:
        pass
    xla_batch_signature = None
    for epoch_num in range(trained_epoch, epoch):
        set_data_loader_epoch(dataset, epoch_num)
        start_time = timer()
        current_epoch = epoch_num

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        
        # Profile first 3 batches of first epoch for FLOPs measurement
        should_profile = current_epoch == trained_epoch and device.type != "xla"
        
        for step, data in enumerate(dataset):
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            if device.type == "xla":
                xla_batch_signature = validate_xla_batch_shape(data, xla_batch_signature)

            optimizer.zero_grad()

            # Profile specific batches
            if should_profile and step < 3:
                activities = [ProfilerActivity.CPU]
                assert device is not None
                if device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                
                with profile(  # noqa: SIM117
                    activities=activities,
                    record_shapes=True,
                    with_flops=True
                ) as prof:
                    with record_function("model_forward"):
                        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp): # type: ignore
                            results = model(data, config, gpu_list, acc_result, "train")
                            loss, acc_result = results["loss"], results["acc_result"]
                
                # Extract FLOPs from profiler
                total_flops = sum([evt.flops for evt in prof.key_averages() if evt.flops > 0])
                profiling_metrics["total_flops"] += total_flops
                profiling_metrics["profiled_batches"] += 1
                
                logger.info("Profiled batch %d: %.2f GFLOPs", step, total_flops / 1e9)
            else:
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp): # type: ignore
                    results = model(data, config, gpu_list, acc_result, "train")
                    loss, acc_result = results["loss"], results["acc_result"]
            
            total_loss += loss.detach().item()

            scaler.scale(loss).backward()
            # Gradient clipping (desescala antes de clipar quando AMP está ativo)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            _optimizer_step(optimizer, scaler, device)
            if warmup_scheduler is not None:
                warmup_scheduler.step()

            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "{}/{}".format(
                    gen_time_str(timer() - start_time), gen_time_str((timer() - start_time) * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", loss.detach().item(), global_step)
        
        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError
        
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(timer() - start_time), gen_time_str((timer() - start_time) * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        checkpoint(str(output_path / f"{current_epoch}.pkl"), model, optimizer, current_epoch, config,
                   global_step, warmup_scheduler=warmup_scheduler, device=device)
        writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1),
                          current_epoch)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_res = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list,
                                 output_function)
                # ── Early Stopping check ────────────────────────────────────────
                if es_patience > 0 and eval_res is not None:
                    val_loss = _distributed_validation_loss(
                        eval_res.get("loss", float("inf")),
                        device,
                        current_epoch,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        es_counter = 0
                    else:
                        es_counter += 1
                        logger.info(
                            "Early stopping: no improvement for %d/%d epochs (best=%.4f, current=%.4f)",
                            es_counter, es_patience, best_val_loss, val_loss,
                        )
                        if es_counter >= es_patience:
                            logger.info("Early stopping triggered at epoch %d", current_epoch)
                            early_stopped = True

        # StepLR só atua quando não há warmup scheduler (outros otimizadores)
        if warmup_scheduler is None:
            exp_lr_scheduler.step()

        if early_stopped:
            break
    
    # Save profiling metrics to file
    if profiling_metrics["profiled_batches"] > 0:
        profiling_metrics["avg_flops_per_batch"] = profiling_metrics["total_flops"] / profiling_metrics["profiled_batches"] # type: ignore
        actual_training_steps = global_step - _initial_global_step

        metrics_path = output_path / "profiling_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "total_flops": profiling_metrics["total_flops"],
                "avg_flops_per_batch": profiling_metrics["avg_flops_per_batch"],
                "profiled_batches": profiling_metrics["profiled_batches"],
                "actual_training_steps": actual_training_steps,
                "total_gflops": profiling_metrics["total_flops"] / 1e9,
                "avg_gflops_per_batch": profiling_metrics["avg_flops_per_batch"] / 1e9,
                "actual_total_gflops": profiling_metrics["avg_flops_per_batch"] / 1e9 * actual_training_steps,
            }, f, indent=2)

        logger.info(
            "Profiling complete: %.2f GFLOPs/batch avg, %d steps → %.1f TFLOPs total",
            profiling_metrics["avg_flops_per_batch"] / 1e9,
            actual_training_steps,
            profiling_metrics["avg_flops_per_batch"] / 1e9 * actual_training_steps / 1000,
        )
