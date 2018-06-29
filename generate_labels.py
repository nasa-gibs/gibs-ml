from datetime import datetime, time, timedelta, date
from dateutil.relativedelta import relativedelta
from utils import daterange

# Start and end dates
start_date = date(2015, 11, 24)
end_date = date(2016, 1, 1)

# Define file 
output_file = "data/labels_1.txt"
with open(output_file, "w") as f:  

	# Loop through dates
	for single_date in daterange(start_date, end_date):
		datestring = single_date.strftime("%Y-%m-%d")
		f.write(datestring + " \n")