import argparse



if __name__ == '__main__':
    args = argparse.ArgumentParser()  # its a class
    args.add_argument("--name", "-n", default = "Neetesh", type = str)
    args.add_argument("--age", "-a", default = 25.0, type = float)
    parse_args = args.parse_args() 

    print(parse_args.name, parse_args.age)