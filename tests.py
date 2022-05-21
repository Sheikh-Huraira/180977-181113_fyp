from Pre_data import PreData
aa = PreData()

aa.Data_pre()
qq=aa.Calculate_tfidf()
with open("tttt.txt", "w", encoding="utf-8") as f:
    f.write(str(qq))
c = aa.test()
print(c)