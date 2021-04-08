for dir in */; do
   cd $dir
   echo "$dir"
   for script in *.py; do
      echo -e "\t-$script"
      python3 $script > "${script%.*}.log"
   done
   cd ..
done
