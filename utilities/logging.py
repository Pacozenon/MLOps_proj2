import logging

class MyLogger:
    def __init__(self, name, level=logging.DEBUG, filename='log_file.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)