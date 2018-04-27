import thulac

thu1 = thulac.thulac(seg_only=True, model_path="thulac/models/", T2S=False, filt=True)


def process_by_line(filename1, filename2):
    print('processing '+ filename1)
    content = ''
    with open(filename1, 'r') as f:
        count = 0
        for line in f.readlines():
            items = line.split('\t')
            news_seg = thu1.cut(items[1], text=True)
            items[1] = news_seg;
            record = ''
            for item in items:
                record = record + item + '\t'
            record = record[:-1]
            content = content + record
            if count % 100 == 0:
                print('processing #' + str(count))
            count = count + 1
    print('writing '+ filename2)
    with open(filename2, 'w+') as f:
        f.write(content)


process_by_line('data/News_info_train_plain.txt', 'data/News_info_train_seg.txt')
process_by_line('data/News_info_validate_plain.txt', 'data/News_info_validate_seg.txt')
