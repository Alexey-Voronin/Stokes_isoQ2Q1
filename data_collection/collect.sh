source ~/firedrake/bin/activate

DIRs=('two_grid' 'multi_grid' 'bfs_scaling')

for dir in ${DIRs[*]}; do
   cd $dir
   echo "$dir"
   for script in *.py; do
      echo -e "\t-$script"
      python3 $script > "${script%.*}.log"
   done
   cd ..
done
