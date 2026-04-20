from model import Tester
from config import make_cfg

def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()

if __name__ == "__main__":
    main()
