#!/bin/bash

# 将FCPCodebase打包成纯正的二进制文件
# 执行前请安装依赖
# pip install nuitka
# sudo apt install patchelf
# 注：nuitka打包

# Call this script from any directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# cd to main folder
cd "${SCRIPT_DIR}/.."

binary_name="AgentSpark"
teamname_name="RoboCup3D-pkg"

rm -rf ./package/Run_Player.build
rm -rf ./package/Run_Player.dist
rm -rf ./package/Run_Player.onefile-build
rm -rf ./package/target

# 生成新的秘钥
python3 encrypt/Encrypt.py generate_key

# bundle app, dependencies and data files into single executable
python3 -m nuitka --lto=no --onefile --standalone --output-dir=package/ --output-filename=${binary_name} Run_Player.py

mkdir -p ./package/target/${teamname_name}/behaviors/
mkdir -p ./package/target/${teamname_name}/world/commons/
mv ./package/${binary_name} ./package/target/${teamname_name}/
cp -r ./behaviors/slot/ ./package/target/${teamname_name}/behaviors/
cp -r ./pkl/ ./package/target/${teamname_name}/
cp -r ./world/commons/robots/ ./package/target/${teamname_name}/world/commons/

cp -r ./libs ./package/target/${teamname_name}/
./package/copy_libs.sh ./package/target/${teamname_name}/${binary_name} ./package/target/${teamname_name}/libs

# 加密打包后的pkl文件
python3 encrypt/Encrypt.py encrypt package/target/${teamname_name}/

# start.sh
cat > ./package/target/${teamname_name}/start.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1
LIBS_DIR=./libs

host=\${1:-localhost}
port=\${2:-3100}

export LD_LIBRARY_PATH=\$LIBS_DIR:\$LD_LIBRARY_PATH

for i in {1..11}; do
  ./${binary_name} -i \$host -p \$port -u \$i -t ${teamname_name} -P 0 -D 0 &
done
EOF

# start_penalty.sh
cat > ./package/target/${teamname_name}/start_penalty.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1
LIBS_DIR=./libs

host=\${1:-localhost}
port=\${2:-3100}

export LD_LIBRARY_PATH=\$LIBS_DIR:\$LD_LIBRARY_PATH

./${binary_name} -i \$host -p \$port -u 1  -t ${teamname_name} -P 1 -D 0 &
./${binary_name} -i \$host -p \$port -u 11 -t ${teamname_name} -P 1 -D 0 &
EOF

# start_fat_proxy.sh
cat > ./package/target/${teamname_name}/start_fat_proxy.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

LIBS_DIR=./libs

host=\${1:-localhost}
port=\${2:-3100}

export LD_LIBRARY_PATH=\$LIBS_DIR:\$LD_LIBRARY_PATH

for i in {1..11}; do
  ./${binary_name} -i \$host -p \$port -u \$i -t ${teamname_name} -F 1 &
done
EOF

# kill.sh
cat > ./package/target/${teamname_name}/kill.sh << EOF
#!/bin/bash
pkill -9 -e ${binary_name}
EOF

# execution permission
chmod a+x ./package/target/${teamname_name}/start.sh
chmod a+x ./package/target/${teamname_name}/start_penalty.sh
chmod a+x ./package/target/${teamname_name}/start_fat_proxy.sh
chmod a+x ./package/target/${teamname_name}/kill.sh

cd ./package/target
tar -zvcf ${teamname_name}.tar.gz ./${teamname_name}