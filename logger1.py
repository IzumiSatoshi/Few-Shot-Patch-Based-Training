import tensorflow as tf
import os
import shutil


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        """Create a summary writer logging to log_dir."""
        writer = tf.summary.create_file_writer(log_dir, filename_suffix=suffix)
        with writer.as_default():
            for step in range(100):
                # other model code would go here
                tf.summary.scalar("my_metric", 0.5, step=step)
                writer.flush()
            
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


class ModelLogger(object):
    def __init__(self, log_dir, save_func):
        self.log_dir = log_dir
        self.save_func = save_func

    def save(self, model, epoch, isGenerator):
        if isGenerator:
            new_path = os.path.join(self.log_dir, "model_%05d.pth" % epoch)
        else:
            new_path = os.path.join(self.log_dir, "disc_%05d.pth" % epoch)
        self.save_func(model, new_path)

    def copy_file(self, source):
        shutil.copy(source, self.log_dir)

