import sys

if(len(sys.argv) < 2):
  print('Plase provide the csv file path to spilt')
  sys.exit()

if(len(sys.argv) == 3):
  splitlevel = sys.argv[2]
else:
  splitlevel = 5

fil = sys.argv[1]
csvfilename = open(fil, 'r', -1).readlines()
file = 1

#Number of lines to be written in new file
record_per_file = int(11000000/int(splitlevel))

for j in range(len(csvfilename)):
  if j % record_per_file == 0:
    write_file = csvfilename[j:j+record_per_file]
    open(str(fil)+ str(file) + '.csv', 'w+').writelines(write_file)
    file += 1
