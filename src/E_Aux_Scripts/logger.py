import logging
import os
import os.path as osp
import datetime

class Logger:
    def __init__(self, experiment_type=None, sub_experiment_type=None, model_type=None, setup=None, base_logs_path='data/logs', log_file='info.log', level=logging.INFO, to_console=True):
        """
        Initialize logger with experiment-specific directory structure.
        
        Args:
            experiment_type: Type of experiment ('DDPM', 'Classifiers' or 'Evaluation')
            sub_experiment_type: Type of sub-experiment ('train', 'test')
            model_type: Type of model being used (e.g., 'convnext', 'vit', 'fpn-mil')
            setup: Type of setup (e.g., 'no_cf', 'cf', ...)
            base_logs_path: Base path for all logs (default: 'data/logs')
            log_file: Name of the log file (default: 'info.log')
            level: Logging level (default: logging.INFO)
            to_console: Whether to also print logs to console (default: True)
        """
        # Build experiment directory structure
        experiment_log_dir = ""
        if experiment_type:
            experiment_log_dir = f"{experiment_type}"
            if sub_experiment_type:
                experiment_log_dir = f"{experiment_log_dir}/{sub_experiment_type}"
            if model_type:
                experiment_log_dir = f"{experiment_log_dir}_{model_type}"
            if setup:
                experiment_log_dir = f"{experiment_log_dir}/{setup}"
        else:
            experiment_log_dir = "Other"

        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        
        if slurm_job_id:
            dir_name = f"job_{slurm_job_id}_{timestamp}"
        else:
            dir_name = f"local_{timestamp}"

        self.output_dir = osp.join(
            base_logs_path,
            experiment_log_dir,
            dir_name,
        )
        
        self.output_dir = os.path.expanduser(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logger
        target_path = os.path.abspath(os.path.join(self.output_dir, log_file))
        # Use a path-scoped logger name to avoid cross-talk between different runs.
        logger_name = f"ModelLogger:{target_path}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S'
        )

        file_handler_exists = any(
            isinstance(handler, logging.FileHandler) and
            os.path.abspath(getattr(handler, 'baseFilename', '')) == target_path
            for handler in self.logger.handlers
        )

        if not file_handler_exists:
            file_handler = logging.FileHandler(target_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Whether to print to terminal
        if to_console:
            stream_handler_exists = any(
                isinstance(handler, logging.StreamHandler) and
                not isinstance(handler, logging.FileHandler)
                for handler in self.logger.handlers
            )
            if not stream_handler_exists:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

    def debug(self, message):
        """Log debug level message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info level message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning level message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error level message"""
        self.logger.error(message)

    def log_training(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                         f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    def log_inference(self, predictions, targets):
        accuracy = (predictions == targets).mean()
        self.logger.info(f'Inference Accuracy: {accuracy:.4f}')

    def log_message(self, message):
        self.logger.info(message)