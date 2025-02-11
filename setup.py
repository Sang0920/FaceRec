import os
from datetime import datetime, timedelta
from api_client import DracoAPIClient
import subprocess
from dotenv import load_dotenv
load_dotenv()
SHIFT_NAME = os.getenv('SHIFT_NAME')

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
    check_in_duration = buffer_before * 60
    check_out_duration = buffer_after * 60
    print(f"Shift timing: {start_time} - {end_time}")
    print(f"Buffer before: {buffer_before} minutes")
    print(f"Buffer after: {buffer_after} minutes")
    print(f"Shift details: {earliest_checkin} - {latest_checkout}")
    print(f"Checkin duration: {check_in_duration}")
    print(f"Checkout duration: {check_out_duration}")
    log_path = "./logs"
    os.makedirs(log_path, exist_ok=True)
    with open('crontab_commands', 'w') as f:
        # f.write(f"# Checkin everyday at {earliest_checkin.strftime('%M %H')} to {start_time.strftime('%M %H')}\n")
        # f.write(f"{earliest_checkin.strftime('%M %H')} * * * * ./main.py --process_duration {check_in_duration} --checkin_type IN >> {log_path}/checkin.log 2>&1\n\n")
        # f.write(f"# Checkout everyday at {end_time.strftime('%M %H')} to {latest_checkout.strftime('%M %H')}\n")
        # f.write(f"{end_time.strftime('%M %H')} * * * * ./main.py --process_duration {check_out_duration} --checkin_type OUT >> {log_path}/checkout.log 2>&1\n")
        
        #22 16 * * * /home/teamdev/FaceRec/testing_app/run_main.sh -d 60 -t IN >> /home/teamdev/FaceRec/testing_app/logs/cron.log 2>&1
        f.write(f"# Checkin everyday at {earliest_checkin.strftime('%M %H')} to {start_time.strftime('%M %H')}\n")
        f.write(f"{earliest_checkin.strftime('%M %H')} * * * /home/teamdev/FaceRec/testing_app/run_main.sh -d {check_in_duration} -t IN > /home/teamdev/FaceRec/testing_app/logs/check-in.log 2>&1\n\n")
        f.write(f"# Checkout everyday at {end_time.strftime('%M %H')} to {latest_checkout.strftime('%M %H')}\n")
        f.write(f"{end_time.strftime('%M %H')} * * * /home/teamdev/FaceRec/testing_app/run_main.sh -d {check_out_duration} -t OUT > /home/teamdev/FaceRec/testing_app/logs/check-out.log 2>&1\n")

        f.write('\n')
    print(f"Generated crontab commands in {os.path.abspath('crontab_commands')}")
    return True

if __name__ == "__main__":
    try:
        generate_crontab_commands()
        # Install the crontab commands
        try:
            result = subprocess.run(['crontab', 'crontab_commands'], check=True)
            print("Successfully installed crontab commands")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install crontab commands: {e}")
            exit(1)
            
    except Exception as e:
        print(f"Error generating crontab commands: {e}")
        exit(1)