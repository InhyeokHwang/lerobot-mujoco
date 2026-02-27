### quest3 setting
cd ~/quest3
mkcert -install

mkcert -cert-file cert.pem -key-file key.pem \
192.168.50.172 localhost 127.0.0.1

server url
https://192.168.xxx.xxx:8012?ws=wss://192.168.xxx.xxx:8012

