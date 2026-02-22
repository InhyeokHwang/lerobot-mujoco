cd ~/quest3
mkcert -install

mkcert -cert-file cert.pem -key-file key.pem \
192.168.219.114 localhost 127.0.0.1

https://192.168.219.114:8012?ws=wss://192.168.219.114:8012