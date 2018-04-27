awk -F '\t' '{print $2}' ../../data/News_info_train_seg.txt > data/X.train
awk -F '\t' '{print $2}' ../../data/News_info_validate_seg.txt > data/X.test

echo "" > data/all_text
cat data/X.train > data/all_text
cat data/X.test >> data/all_text

awk -F '\t' '{print $2}' ../../data/News_pic_label_train.txt > data/Y.train
awk -F '\t' '{print $1}' ../../data/News_info_validate_seg.txt > data/id.test
echo "Done!"



