#!/usr/bin/env python3

import argparse, sys
from run import train, test

def getargs():
    parser = argparse.ArgumentParser(description='Train CNN model with Hi-C data and make predictions for low resolution HiC data.')

    # Adding Subparsers
    subparsers = parser.add_subparsers(dest='subcommands')

    # Parser for Training command
    subtrain = subparsers.add_parser('train',
            help='''Train CNN model''')
    subtrain.set_defaults(func=train)

    # Parser for Predicting Command
    subchrom = subparsers.add_parser('pred_chromosome',
            help='''predict high resolution interaction frequencies for inter and intra chromosomes''')
    subchrom.set_defaults(func=test)

    
    # Adding Arguments for Train
    subtrain.add_argument('-i', '--inputfile',
                        help = 'path to a .hic file.', type = str)
    subtrain.add_argument('-e', '--epochs',
                        help = 'Number of Epochs to train model',
                        type = int, default = 10)
    subtrain.add_argument('-b', '--batch size',
                        help = 'Batch Size used by model to train',
                        type = int, default = 250)

    commands = sys.argv[1:]
    if ((not commands) or ((commands[0] in ['train', 'pred_chromosome'])
        and len(commands) == 1)):
        commands.append('-h')
    args = parser.parse_args(commands)

    return args, commands

# args = parser.parse_args()
# print(args.accumulate(args.integers))


def run():
    
    print('Parseing Arguments')
    args, commands = getargs()

    if commands[0] not in ['-h','--help']:
        args.func(args)


if __name__ == '__main__':
    run()
