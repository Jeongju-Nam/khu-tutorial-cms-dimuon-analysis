import uproot

# string
path = "/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root"
print(path)
print('type of path: ', type(path))

file = uproot.open(path)
print(file)

print(file.keys())

tree = file['Events']
print(tree)

print('Number of Events:')
print(tree.num_entries)

tree_iter = tree.iterate(step_size=1, library='np')
print(tree_iter)

#event = next(tree_iter)
#print('First event: ')
#print(event)

#event = next(tree_iter)
#print("second: ")
#print(event)

count = 0

# for loop
for event in tree_iter:
    print(event)
    print('---------')

    #count = count + 1
    count += 1

    if count >= 5:
        break

print('count: ', count)
print('we break the loop')

print('type of event: ', type(event))
