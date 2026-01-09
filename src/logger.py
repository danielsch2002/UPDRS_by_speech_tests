import logging
import sys

"""
Logging Configuration Module
----------------------------
This section initializes the project-wide logger. 
It uses a StreamHandler to direct output to the console (stdout) 
and a Formatter to ensure all logs include timestamps and severity levels.
"""

# 1. Initialize the logger with a meaningful name
logger_inst = logging.getLogger("UPDRS_Project")

# 2. Set the minimum logging level to INFO
# This ensures that info, warning, error, and critical logs are displayed.
logger_inst.setLevel(logging.INFO)

# 3. Define the output destination (Console)
# sys.stdout is used to ensure logs appear in the standard output stream.
console_handler = logging.StreamHandler(sys.stdout)

# 4. Define a professional log format
# Format: Year-Month-Day Hour:Minute:Second - Name - Level - Message
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 5. Apply the formatter to the handler and add the handler to the logger
console_handler.setFormatter(log_formatter)
logger_inst.addHandler(console_handler)

"""
Note: We do not wrap this in a class because the logging module 
is designed to be used as a global singleton across the package.
"""