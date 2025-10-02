# HƯỚNG DẪN ĐO CPU USAGE TRÊN USERSPACE

## 1. Truy cập server3 bằng tailscale
- Gửi gmail để được add vào node tailscale
- Sử dụng lệnh sau để ssh vào server3: 
```bash
ssh dongtv@<IP server3>
```
- Mật khẩu server3 và lanforge: 1   

**BẮT ĐẦU TỪ ĐÂY, MỞ 2 TAB LÀM VIỆC**
## 2. Switch giữa các branch trong code (Làm ở VS code):
- Truy cập vào server3, kiểm tra xem đang nằm ở branch nào:
```bash
cd dtuan/xdp-program
git branch
```
```bash
git switch userspace
```
```bash
make
```
## 3. Thực hiện trên terminal như sau:
- Mở một terminal ở máy tính, ssh vào server3 với câu lệnh ssh như ở phần 1.
- Sau khi ssh vào, gõ lệnh sau:
```bash
byobu
```
Sau khi gõ lệnh trên, màn hình hiển thị như sau:
![Blabla](img/estimateblabla.png)

- Góc phải dưới chạy lệnh:
```bash
sudo tcpdump -i eno3
```
- Click chuột vào ô góc trái bên dưới, chạy lệnh sau:
```bash
./dtuan/run_xdp_program.sh
```
- Chạy test trên userspace ở ô bên trái trên, như sau:
```bash
cd userspace_test
source .venv/bin/active
sudo taskset -c 1 .venv/bin/python3 main.py --model <model muốn test>
```
+ NOTE: model ở đây có lof, knn, isoforest, randforest
- Tiếp theo đến góc phải trên, chạy lệnh sau:
```bash
./scripts_tcpreplay.sh enp1s0f1 pcap/data_portmap.pcap <số pps> 120 cpu_usage_userspace/<Tên model>.log
```
Số pps thay đổi từ 10000 đến 100000 mỗi lần tăng 10000
## Đo mỗi tham số 5 lần
## Tóm tắt cách đo như sau:
1. Đổi branch -> Điều chỉnh tham số -> make
2. Chạy scripts run_xdp_program
3. Chạy main.py
4. Phát tấn công bằng ./scripts_tcpreplay.sh
5. Chờ phát hết tấn công thì làm lại từ bước 2.
