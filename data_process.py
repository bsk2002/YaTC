import os
import glob
import binascii
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import numpy as np

def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_MFR_bytes(pcap_dir):
    # IP Address virtualization
    ip_map = {}
    virtual_ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    ip_idx = 0

    packets = scapy.rdpcap(pcap_dir)
    data = []
    for packet in packets:
        if not packet.haslayer('IP'):
            continue
        
        ip_layer = packet['IP']

        # 1. IP 일관성 유지 (방향 정보 보존)
        for attr in ['src', 'dst']:
            orig_ip = getattr(ip_layer, attr)
            if orig_ip not in ip_map:
                # 새로운 IP가 등장하면 가상 IP 할당
                if ip_idx < len(virtual_ips):
                    ip_map[orig_ip] = virtual_ips[ip_idx]
                    ip_idx += 1
                else:
                    # 가상 IP 풀이 부족할 경우 대비 (일반적인 1:1 통신에선 발생 드묾)
                    ip_map[orig_ip] = f"10.0.0.{ip_idx + 1}"
                    ip_idx += 1
            
            # 실제 패킷 필드 수정
            setattr(ip_layer, attr, ip_map[orig_ip])
        
        # 2. Port 번호 0으로 변경 (서비스 정보 마스킹)
        if packet.haslayer('TCP'):
            packet['TCP'].sport = 0
            packet['TCP'].dport = 0
        elif packet.haslayer('UDP'):
            packet['UDP'].sport = 0
            packet['UDP'].dport = 0
        
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5-len(data)):
            data.append(('0'*160, '0'*480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data

def MFR_generator(flows_pcap_path, output_path):
    flows = glob.glob(flows_pcap_path + "/*/*/*.pcap")
    makedir(output_path)
    makedir(output_path + "/train")
    makedir(output_path + "/test")
    makedir(output_path + "/valid")
    classes = glob.glob(flows_pcap_path + "/*/*")
    for cla in tqdm(classes):
        makedir(cla.replace(flows_pcap_path, output_path))
    for flow in tqdm(flows):
        content = read_MFR_bytes(flow)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
        fh = np.reshape(content, (40, 40))
        fh = np.uint8(fh)
        im = Image.fromarray(fh)
        im.save(flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path))

if __name__ == '__main__':
    flows_pcap_path = "/home/sangkyoung/Desktop/captured"
    output_path = "."
    
    MFR_generator(flows_pcap_path, output_path)