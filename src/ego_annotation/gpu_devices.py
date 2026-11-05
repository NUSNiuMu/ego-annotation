"""Resolve a local device to its physical UUID before importing CUDA."""
import csv
import os
import subprocess


def inventory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,pstate', '--format=csv,noheader'],
        text=True, timeout=10)
    devices = {}
    for row in csv.reader(output.splitlines()):
        index, uuid, state = (part.strip() for part in row)
        if not uuid.startswith('GPU-'):
            raise RuntimeError('Cannot identify physical GPU: ' + index)
        devices[int(index)] = {'uuid': uuid, 'error': '' if state in {
            'P' + str(i) for i in range(16)} else state}
    if not devices:
        raise RuntimeError('No GPU inventory returned')
    return devices


def device(gpu):
    value = inventory()[gpu]
    if value['error']:
        raise RuntimeError(f'GPU{gpu} unavailable: {value["error"]}')
    return value


def configure(gpu):
    uuid = device(gpu)['uuid']
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    return uuid
