#generate features
awk -F '\t' 'NR == FNR{a[$1] = $2} NR != FNR {if(a[$1] >= 1) print $2}' ../../data/News_pic_label_train.txt ../../data/News_info_train_seg.txt > data/temp1
#remove stopwords
awk -F " " 'NR==FNR {s[$1] = 1} NR != FNR {b="";for(i=1;i<=NF;i++) if(s[$i] !=1 ) b=b""$i" "; print b}' stopwords.txt data/temp1 > data/temp2
#features
awk -F " " '{for(i=1;i<=NF;i++) { a[$i]++}} END{for (i in a) print i, a[i]}' data/temp2 | sort -r -n -k 2 | head -500 > data/features

awk -F '\t' '{print $2}' ../../data/News_info_train_seg.txt > data/temp3

awk -F '\t' '{print $2}' ../../data/News_info_validate_seg.txt > data/temp4

awk 'NR == FNR {idx[FNR] = $1} NR!=FNR { if(feature_size <= 0) feature_size = NR - 1; delete a; for(i=1;i<=NF;i++) {a[$i]++;} b="";for(j=1;j<=feature_size;j++) {if(a[idx[j]]) {b = b " "a[idx[j]]} else {b = b " "0} } print b}' data/features data/temp3 > data/X.train

awk 'NR == FNR {idx[FNR] = $1} NR!=FNR { if(feature_size <= 0) feature_size = NR - 1; delete a; for(i=1;i<=NF;i++) {a[$i]++;} b="";for(j=1;j<=feature_size;j++) {if(a[idx[j]]) {b = b " "a[idx[j]]} else {b = b " "0} } print b}' data/features data/temp4 > data/X.test

awk -F '\t' '{print $2}' ../../data/News_pic_label_train.txt > data/Y.train

rm data/temp*

awk -F '\t' '{print $1}' ../../data/News_info_validate_seg.txt  > data/id.test

echo 'Done!'
