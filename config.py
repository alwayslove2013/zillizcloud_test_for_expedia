def get_config():
    config = {}
    with open("./config.txt", "r") as file:
        for line in file:
            if "#" in line:
                key, value = line.strip().split("#", 1)
                config[key] = value

    return config
