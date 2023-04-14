def main():
    f = open("annotations.csv", "w")

    for i in range(0, 3184):
        f.write("image-"+'{0:010d}'.format(i)+".png,0"+"\n")

    for i in range(8760, 28790):
        f.write("image-" + '{0:010d}'.format(i) + ".png,1" + "\n")
    f.close()

if __name__ == "__main__":
    main()