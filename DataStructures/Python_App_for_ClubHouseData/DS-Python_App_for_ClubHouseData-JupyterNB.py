# -*- coding: utf-8 -*-
"""
DSAD Assignment 1 - Group 166

"""
def initializeHash():
    hash_table = [[] for _ in range(599)]
    return hash_table

def hashID(name):
    ascii_name = 0
    ascii_c = 0
    # generate hash value/index by taking ASCII value of each character, 
    # multiplying by its position in the string and aggregating for all characters
    for idx, c in enumerate(name): 
        ascii_c = ord(c) 
        ascii_name = ascii_name + ascii_c * (idx+1)
    hash_index = ascii_name % len(ApplicationRecords)
    return hash_index

def insertAppDetails(ApplicationRecords, name, phone, memRef, status):
    # convert the name to title case before inserting and generating hash key
    hash_index = hashID(name.title())
    Applicant = [name.title(),phone,memRef,status]
    ApplicationRecords[hash_index].append(Applicant)
    return

def updateAppDetails(ApplicationRecords, name, phone, memRef, status):
    # convert name to title case before generating hash key and searching
    hash_index = hashID(name.title())
    name_found = False
    
    # iterate through individual records stored at same index as the generated hash key    
    # compare names for each record with name from prompt file
    for idx, item in enumerate(ApplicationRecords[hash_index]):
        if item[0] == name.title():
            name_found = True
            updated_fields = ""
            if item[1] != phone: 
                updated_fields+= "Phone"
            if item[2] != memRef: 
                updated_fields = updated_fields + ", " + "memRef"
            if item[3] != status: 
                updated_fields = updated_fields + ", " + "Status"
        # If no fields have changed, write message to Output file
            if updated_fields == "":
                output_file.write("\n\nUpdate Prompt: No changes detected for applicant " + name)
            else:
        # If any field is changed, update hash table and write updates to Output File
                item = [name,phone,memRef,status]
                ApplicationRecords[hash_index][idx] = item
                updated_fields = updated_fields.lstrip(", ")
                output_file.write("\n\nUpdated details of "+ name +". " + updated_fields + " updated")
    # If name given against Update tag is not found, write message
    if name_found == False: output_file.write("\n\nUpdate Error: Applicant " + name + " not found\n")
    return

def memRef(ApplicationRecords, memID):
    ref_count = 0
    output_file.write("\n\n-----Members referred by " + memID + "------")
    
    for bucket in range(len(ApplicationRecords)):
        for idx in range(len(ApplicationRecords[bucket])):
            item = ApplicationRecords[bucket][idx]
            if item[2] == memID:
                ref_count = ref_count + 1
                output_file.write("\n"+item[0]+" / "+item[1]+" / "+item[3])
    if ref_count == 0:
        output_file.write("\nNo applications referred by this member ID")
    output_file.write("\n---------------------------------------")
    return

def appStatus(ApplicationRecords):
    count_applied = 0
    count_verified = 0
    count_approved = 0
        
    for bucket in range(len(ApplicationRecords)):
        for idx in range(len(ApplicationRecords[bucket])):
            if ApplicationRecords[bucket][idx][3] == "Applied":
                count_applied = count_applied + 1
            if ApplicationRecords[bucket][idx][3] == "Verified":
                count_verified = count_verified + 1        
            if ApplicationRecords[bucket][idx][3] == "Approved":
                count_approved = count_approved + 1          
    
    output_file.write("\n\n-------Application Status ---------")
    output_file.write("\nApplied "+str(count_applied))
    output_file.write("\nVerified "+str(count_verified))
    output_file.write("\nApproved "+str(count_approved))
    output_file.write("\n----------------------------------")   
    return

def destroyHash(ApplicationRecords):
    ApplicationRecords.clear()
    return
    
""" Main Program begins """

""" Initialize variables """

input_file_name = "inputPS26.txt" 
output_file_name = "outputPS26.txt"
prompt_file_name = "promptsPS26.txt" 

input_file = open(input_file_name, "r")
output_file = open(output_file_name,"w")
prompt_file = open(prompt_file_name, "r")

ApplicationRecords = initializeHash()

""" Read each line from input file and strip trailing newline character
    Split using / as separator, remove leading and trailing whitespaces """

app_count = 0
input_lines = (line.rstrip() for line in input_file)
for x in input_lines:
    if x: 
        a = x.split("/")
        a = [y.strip(' ') for y in a] 
        insertAppDetails(ApplicationRecords, a[0],a[1],a[2],a[3])
        app_count += 1
    else: continue

output_file.write("Successfully inserted "+str(app_count)+" applications into the system")

""" Read each line from prompt file and check for tags """
    
for x in prompt_file:
    x=x.rstrip()
    if x[:6] == "Update":
        a = x[8:].split("/")
        a = [y.strip(' ') for y in a]
        updateAppDetails(ApplicationRecords, a[0],a[1],a[2],a[3])
   
    if x[:9] == "memberRef":
        memID = x[11:]
        memRef(ApplicationRecords,memID)
    
    if x[:9] == "appStatus":
        appStatus(ApplicationRecords)
   
input_file.close()    
prompt_file.close()
output_file.close()

destroyHash(ApplicationRecords)


 
 
