result_dir=$1
detector=$2

for file in $result_dir/$detector*; do
  res=$(tail -n 1 $file);
  echo "${file} ${res}";
done;

