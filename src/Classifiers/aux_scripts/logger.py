import logging
import os
import os.path as osp
import datetime

class Logger:
    @classmethod
    def configure(cls, experiment_type=None, base_logs_path='data/logs'):
        """
        Configure logger with experiment-specific directory structure.
        
        Args:
            experiment_type: Type of classifier experiment ('convnext', 'vit')
            base_logs_path: Base path for all logs (default: 'data/logs')
            
        Returns:
            str: Path to the configured log directory
        """
        if experiment_type == "convnext":
            experiment_log_dir = "Classifier_logs/ConvNeXt"
        elif experiment_type == "vit":
            experiment_log_dir = "Classifier_logs/ViT"
        else:
            experiment_log_dir = "other"
        
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        
        dir_name = f"job_{slurm_job_id}_{timestamp}"
        
        log_dir = osp.join(
            base_logs_path,
            experiment_log_dir,
            dir_name,
        )
        
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        return log_dir
    
    def __init__(self, log_dir='logs', log_file='training.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('ModelLogger')
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def debug(self, message):
        """Log debug level message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info level message"""
        self.logger.info(message)
    
    def warn(self, message):
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