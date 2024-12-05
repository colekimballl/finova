import json
import time
from datetime import datetime, timedelta
from termcolor import cprint
import random


def load_tasks():
    with open("/Users/colekimball/ztech/finova/productivity/tasks.json", "r") as f:
        tasks = json.load(f)
    return tasks


def get_tasks_schedule(tasks):
    task_start_time = datetime.now()
    schedule = []
    for task, minutes in tasks.items():
        end_time = task_start_time + timedelta(minutes=minutes)
        schedule.append((task, task_start_time, end_time))
        task_start_time = end_time
    return schedule


def main():
    tasks = load_tasks()
    schedule = get_tasks_schedule(tasks)
    current_index = 0

    while True:
        now = datetime.now()
        current_task, start_time, end_time = schedule[current_index]
        remaining_time = end_time - now
        remaining_minutes = int(remaining_time.total_seconds() // 60)

        print("")

        for index, (task, s_time, e_time) in enumerate(schedule):
            if index < current_index:
                print(f'{task} done: {e_time.strftime("%H:%M")}')
            elif index == current_index:
                if remaining_minutes < 2:
                    cprint(f"{task} < 2m left!", "white", "on_red", attrs=["blink"])
                elif remaining_minutes < 5:
                    cprint(f"{task} - {remaining_minutes} mins", "white", "on_red")
                else:
                    cprint(f"{task} - {remaining_minutes} mins", "white", "on_blue")
            else:
                print(f'{task} @ {s_time.strftime("%H:%M")}')

        if remaining_minutes <= 0:
            current_index += 1
            if current_index >= len(schedule):
                break
        list_of_reminders = [
            "Take a break!",
            "Stay hydrated!",
            "Remember to stretch!",
            "Keep up the good work!",
            "Don't forget to eat!",
            "Stay focused!",
            "You're doing great!",
            "Keep pushing through!",
            "Take a deep breath!",
            "Stay positive!"
            "Jobs not finished"
            "I become my thoughts"
            "I am a good person"
            "I will become the person I want to be"
            "Time is irrelevant"
            "He is with you"
            "This is for you and your family"
            "Rest at the end"
            "Momba Mentality"
            "Deal was already made"
            "Positive thoughts",
        ]
        random_reminder = random.choice(list_of_reminders)
        print(random_reminder + " :D")

        if now >= end_time:
            current_index += 1
            if current_index >= len(schedule):
                cprint("All tasks are completed!", "white", "on_green")


if __name__ == "__main__":
    main()
