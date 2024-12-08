import os
import re
import requests

# 设置搜索词和要下载的图片数量
search = '猫'
total = 100
# 创建一个保存图片的文件夹
if not os.path.exists(search):
    os.mkdir(search)
# 设置请求头
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'}


# 定义下载图片的函数
def load_images():
    tmp = 0
    page = 1
    while True:
        page_url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=10371129381236677678&ipn=rj&ct=201326592&is=&fp=result&fr=&word={}&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&expermode=&nojc=&isAsync=&pn={}&rn=30&gsm=3c&1682846532783='.format(
            search, search, page * 30)
        # 获取网页内容
        res = requests.get(page_url, headers=headers).text
        # 提取图片链接
        url_list = re.findall('"thumbURL":"(.*?)"', res)
        url_set = set()
        for url in url_list:
            if tmp == total:
                break
            u = re.findall('u=(.*?),', url)[0]
            if u not in url_set:
                url_set.add(u)
                # 获取图片
                img = requests.get(url, headers=headers).content
                # 将图片保存到本地
                with open('{}/{}.jpg'.format(search, tmp), 'wb') as f:
                    f.write(img)
                    print(str(tmp) + '.jpg')
                    tmp += 1
        if tmp == total:
            break
        page += 1


if __name__ == '__main__':
    load_images()
