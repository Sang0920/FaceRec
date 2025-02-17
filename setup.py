import os
from datetime import datetime, timedelta
from api_client import DracoAPIClient
import subprocess
from dotenv import load_dotenv
load_dotenv()
SHIFT_NAME = os.getenv('SHIFT_NAME')
CRONTAB_FILE = 'crontab_commands'

def generate_crontab_commands():
    """Generate crontab commands based on shift details"""
    client = DracoAPIClient()
    shift_details = client.get_shift_details(shift_name=SHIFT_NAME)
    if not shift_details.get('success'):
        raise ValueError(f"Failed to get shift details: {shift_details.get('message')}")
    timing = shift_details['shift_timing']
    start_time = datetime.combine(datetime.today(), datetime.strptime(timing['start_time'], '%H:%M:%S').time())
    end_time = datetime.combine(datetime.today(), datetime.strptime(timing['end_time'], '%H:%M:%S').time())
    buffer_before = int(timing['begin_check_in_before_shift_start_time'])
    buffer_after = int(timing['allow_check_out_after_shift_end_time'])
    earliest_checkin = start_time - timedelta(minutes=buffer_before)
    latest_checkout = end_time + timedelta(minutes=buffer_after)
    check_in_duration = buffer_before * 60 # Convert to seconds
    check_out_duration = buffer_after * 60
    print(f"Shift timing: {start_time} - {end_time}")
    print(f"Buffer before: {buffer_before} minutes")
    print(f"Buffer after: {buffer_after} minutes")
    print(f"Shift details: {earliest_checkin} - {latest_checkout}")
    print(f"Checkin duration: {check_in_duration}")
    print(f"Checkout duration: {check_out_duration}")
    dir = os.path.dirname(os.path.realpath(__file__))
    log_path = "./logs"
    os.makedirs(log_path, exist_ok=True)
    if os.path.exists('custom_crontab'):
        with open('custom_crontab', 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2: # Need at least 2 lines to install crontab
                return "exists"
            else:
                print("Custom crontab file exists but does not have at least 2 lines")
                print("Please add the crontab commands to the file and try again")
                return False
    with open(CRONTAB_FILE, 'w') as f:
        f.write(f"# Check-in everyday at {earliest_checkin.strftime('%H:%M')} to {start_time.strftime('%H:%M')}\n")
        f.write(f"{earliest_checkin.strftime('%M %H')} * * * {dir}/run_main.sh -d {check_in_duration} -t IN > {dir}/logs/check-in.log 2>&1\n\n")
        f.write(f"# Check-out everyday at {end_time.strftime('%H:%M')} to {latest_checkout.strftime('%H:%M')}\n")
        f.write(f"{end_time.strftime('%M %H')} * * * {dir}/run_main.sh -d {check_out_duration} -t OUT > {dir}/logs/check-out.log 2>&1\n\n")
    print(f"Generated crontab commands in {os.path.abspath(CRONTAB_FILE)}")
    return True

if __name__ == "__main__":
    try:
        result = generate_crontab_commands()
        if result == "exists":
            print("Custom crontab file exists.")
            subprocess.run(["crontab", "custom_crontab"], check=True)
            print("✅ Custom crontab commands installed successfully")
        elif result:
            print("Installing crontab commands")
            subprocess.run(f"crontab {CRONTAB_FILE}", shell=True)
            print("✅ Crontab commands installed successfully")
        else:
            print("❌ Crontab commands not installed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")