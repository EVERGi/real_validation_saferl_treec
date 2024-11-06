import os

houses_dir = "data/houses/"

house_dirs = [f"{houses_dir}house_{i}/" for i in [1, 2, 3, 5]]

to_replace = {
    '"start time"': '"start_time"',
    '"end time"': '"end_time"',
    '"time step"': '"time_step"',
    '"number of nodes"': '"number_of_nodes"',
    '"node(s) number"': '"node(s)_number"',
    '"Environment 0"': '"Environment_0"',
    '"node number"': '"node_number"',
    '"node(s)_number"': '"nodes_number"',
}

for i in range(6):
    to_replace[f'"Asset {i}"'] = f'"Asset_{i}"'


dates_to_do = ["2024-01-01_0000_2024-04-01_0000", "2024-04-08_1500_2024-06-17_1500"]
for house_dir in house_dirs:
    for date in dates_to_do:
        date_dir = f"{house_dir}{date}/"
        for file in os.listdir(date_dir):
            if file.endswith(".json"):
                with open(f"{date_dir}{file}", "r") as f:
                    text = f.read()

                for key, value in to_replace.items():
                    text = text.replace(key, value)

                with open(f"{date_dir}{file}", "w") as f:
                    f.write(text)
