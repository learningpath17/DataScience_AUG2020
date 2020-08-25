# Example to connect myDB
import mysql.connector
conn = mysql.connector.connect(user='root',password='learningpath',host='localhost')
mycursor=conn.cursor()
mycursor.execute('SHOW TABLES')
print(mycursor.fetchall())


# database connector.

import mysql.connector
conn = mysql.connector.connect(user='root',password='learningpath',host='localhost')
mycursor=conn.cursor()
  
# Create Database 
mycursor.execute('CREATE DATABASE test2')
mycursor.execute('USE test2')

mycursor.execute('SHOW TABLES')
print(mycursor.fectchall())

# Create table in specified Database.
mycursor.execute('USE test2')
mycursor.execute('''CREATE TABLE student
                 (
                 Rno int primary key,
                 name varchar(30) ,
                 gender char(1) ,
                 age int ,
                 email varchar(30) ,
                 city varchar(25) 
                  ''')
mycursor.execute('SHOW TABLES')
print(mycursor.fetchall())

# Insert rows into table
mycursor.execute('''INSERT INTO customer VALUES
                 (1,'ABHI','M',7,'abhi.esuraju@gmail.com','Hyderabad','2018-01-01') ''')
mycursor.execute('''INSERT INTO customer VALUES
                 (2,'Rithwik','M',3,'rith.esuraju@gmail.com','Hyderabad','2018-01-01') ''')
mycursor.execute('''INSERT INTO customer VALUES
                 (3,'venkat','M',31,'venkat.esuraju@gmail.com','Hyderabad','2018-01-01') ''')

conn.commit()

mycursor.execute('select * from customer')
print(mycursor.fetchall())             
# row by row fecthing
mycursor.execute('select * from customer')
mylist=mycursor.fetchall()                         
for x in mylist:
    print(x)
    print('\n')
    
# Update MySQL table
mycursor.execute('update customer set age=99 where id=2')
mycursor.execute('select * from customer')
mylist=mycursor.fetchall()                         
for x in mylist:
    print(x)
    print('\n')

conn.commit()

#delete a row from table
mycursor.execute('delete from customer where id=2')
mycursor.execute('select * from customer')
mylist=mycursor.fetchall()                         
for x in mylist:
    print(x)
    print('\n')

conn.commit()    