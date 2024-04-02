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
def read_rdf():
    from rdflib import Graph

    # 加载 RDF 数据文件
    g = Graph()
    g.parse("your_rdf_data.ttl", format="turtle")

    # 定义 SPARQL 查询
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object .
    }
    LIMIT 10
    """

    # 执行 SPARQL 查询
    results = g.query(query)

    # 输出查询结果
    for row in results:
        print(row)


if __name__ == '__main__':
    read_rdf()