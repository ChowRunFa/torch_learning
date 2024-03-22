def generate_txt_label():

    import os

    root_dir = 'dataset/hymenoptera_data/train'

    for target_dir,out_dir in ['ants_image','ants_label'],['bees_image','bees_label']:

        img_path = os.listdir(os.path.join(root_dir, target_dir))
        label = target_dir.split('_')[0]

        label_dir = os.path.join(root_dir, out_dir)

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        for i in img_path:
            file_name = i.split('.jpg')[0]
            with open(os.path.join(root_dir, out_dir,"{}.txt".format(file_name)),'w') as f:
                f.write(label)

if __name__ == '__main__':
    generate_txt_label()