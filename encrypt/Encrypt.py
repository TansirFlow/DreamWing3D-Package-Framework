from cryptography.fernet import Fernet
import os
import base64
import pickle
import sys
import random
from datetime import datetime

class Encrypt():
    KEY="Ux6fEKYPcTzZ9htsdy5PFZcGsTptXXp2cyfIVjmp_zg="

    def __init__(self):
        pass
    
    @staticmethod
    def get_key():
        return Encrypt.KEY

    @staticmethod
    def generate_fernet_key():
        # 生成32字节的随机密钥
        key = os.urandom(32)
        # 将密钥编码为URL安全的Base64格式
        encoded_key = base64.urlsafe_b64encode(key)
        # 返回解码为utf-8字符串的密钥，方便存储或显示
        return encoded_key.decode('utf-8')
    
    @staticmethod
    def update_key_in_file(file_path=os.path.abspath(__file__)):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = file.read()
        
        start_marker = "KEY=\""
        end_marker = "\"\n"
        start_index = file_data.find(start_marker) + len(start_marker)
        end_index = file_data.find(end_marker, start_index)
        file_data = file_data[:start_index] + Encrypt.generate_fernet_key() + file_data[end_index:]

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_data)

    @staticmethod
    def encrypt(plain_text):
        cipher_suite = Fernet(Encrypt.get_key().encode('utf-8'))
        cipher_text = cipher_suite.encrypt(plain_text)
        return cipher_text
    
    @staticmethod
    def decrypt(cipher_text):
        if not isinstance(cipher_text,bytes):
            cipher_text=cipher_text.encode('utf-8')
        cipher_suite = Fernet(Encrypt.get_key().encode('utf-8'))
        plain_text = cipher_suite.decrypt(cipher_text)
        return plain_text
    
    @staticmethod
    def encrypt_pkl(origin,target):
        with open(origin, 'rb') as f:
            plain_text = f.read()
        cipher_text = Encrypt.encrypt(plain_text)
        with open(target, 'wb') as target_f:
            target_f.write(cipher_text)

    @staticmethod
    def decrypt_pkl(origin,target):
        with open(origin, 'rb') as f:
            cipher_text = f.read()
        plain_text = Encrypt.decrypt(cipher_text)
        class_obj = pickle.loads(plain_text)
        with open(target, 'wb') as target_f:
            pickle.dump(class_obj, target_f)

    @staticmethod
    def scan_file(dir_path,suffix):
        pkl_list=[]
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(suffix):
                    pkl_list.append(os.path.join(root, file))
        return pkl_list
    
    @staticmethod
    def encrypt_xml(origin,target):
        with open(origin, 'r') as f:
            plain_text = f.read()
        cipher_text = Encrypt.encrypt(plain_text.encode('utf-8'))
        with open(target, 'w') as target_f:
            target_f.write(cipher_text.decode('utf-8'))

    @staticmethod
    def decrypt_xml(origin,target):
        with open(origin, 'r') as f:
            cipher_text = f.read()
        plain_text = Encrypt.decrypt(cipher_text.encode('utf-8'))
        with open(target, 'w') as target_f:
            target_f.write(plain_text.decode('utf-8'))

    @staticmethod
    def pkg_encrypt_pkl(path="./package/target/Pkl"):
        pkl_list=Encrypt.scan_file(path,".pkl")
        for pkl in pkl_list:
            print(f"encrypt：{pkl}")
            Encrypt.encrypt_pkl(pkl,pkl)
    
    @staticmethod
    def pkg_encrypt_xml(path="./package/target/behaviors"):
        xml_list=Encrypt.scan_file(path,".xml")
        for xml in xml_list:
            print(f"encrypt：{xml}")
            Encrypt.encrypt_xml(xml,xml)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("please run script as python encrypt/Encrypt.py action param")
    else:
        if sys.argv[1]=="generate_key":
            Encrypt.update_key_in_file()
        elif sys.argv[1]=="encrypt":
            if len(sys.argv)<3:
                binary_pkg_root_dir="package/target/RoboCUp3d-pkg"
            else:
                binary_pkg_root_dir=sys.argv[2]
            print(f"encrypt xml and pkl，root_dir：{binary_pkg_root_dir}")
            Encrypt.pkg_encrypt_xml(os.path.join(binary_pkg_root_dir,"behaviors"))
            Encrypt.pkg_encrypt_pkl(os.path.join(binary_pkg_root_dir,"pkl"))
        else:
            print("invalid param")
        
    
