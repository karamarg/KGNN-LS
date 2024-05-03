import subprocess


if __name__ == '__main__':
    print('auc \t f1 \t pr@1 \t pr@3 \t pr@10 \t rec@1 \t rec@3 \t rec@10')
    for _ in range(30):
        proc = subprocess.Popen(f'python main.py', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ret = proc.communicate()[-2].decode().split('\n')[-2]
        print(ret)
