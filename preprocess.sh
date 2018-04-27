if [ ! -d "data" ]; then
  mkdir data
fi
echo "unzip data..."
unzip raw_data/data.zip
echo "preprocessing..."
sed 's/<[^>]*>//g;s/ //g' raw_data/News_info_train.txt > data/News_info_train_plain.txt
sed 's/<[^>]*>//g;s/ //g' raw_data/News_info_validate.txt > data/News_info_validate.txt_plain.txt
echo "tokenizing..."
python word_seg.py
cp raw_data/News_pic_label_train.txt data/
echo "Done!"
