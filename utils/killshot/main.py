def main():
    f = open("annotations.csv", "w")
    t = open("annotations-tests.csv","w")

    """for i in range(0, 3184):
        f.write("image-"+'{0:010d}'.format(i)+".png,0"+"\n")

    for i in range(8760, 28790):
        f.write("image-" + '{0:010d}'.format(i) + ".png,1" + "\n")
    f.close()"""
    for i in range(0, 300):
        f.write("image-" + '{0:010d}'.format(i) + ".png,0" + "\n")

    for i in range(8760, 9760):
        f.write("image-" + '{0:010d}'.format(i) + ".png,1" + "\n")
    f.close()

    for i in range(301, 600):
        t.write("image-" + '{0:010d}'.format(i) + ".png,0" + "\n")

    for i in range(10000, 16000):
        t.write("image-" + '{0:010d}'.format(i) + ".png,1" + "\n")
    t.close()




if __name__ == "__main__":
    main()