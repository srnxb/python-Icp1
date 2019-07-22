class Hotel:
    def __init__(self, hotnam):  # constructor
        self.hotelname = hotnam


class Employee:
    def __init__(self, enam, eid, essn):  # constructor
        self.employeename = enam
        self.empid = eid

    def __getssn(self):  # private data member
        #        print("Employee SSN is:" + self.essn)
        print("Employee SSN is 78787")
        return

    def callprivate(self):
        self.__getssn()

    def getEmployeeDetails(self):
        print("Employee Name:" + self.employeename)
        print("Employee Id is" + self.empid)


class Room:
    def __init__(self, rnum, rtype):
        self.roomnum = rnum
        self.roomtyp = rtype

    def GetRoomDetails(self):
        print("Room number:" + self.roomnum)
        print("Room Type Allocated for you is" + self.roomtyp)


class Occupants:
    def __init__(self, onam, ophon, oid):
        self.occname = onam
        self.occphone = ophon
        self.occid = oid

    def GetOccupantDetails(self):
        print("Guest number:" + self.occname)
        print("Guest Phone numbber" + self.occphone)
        print("Guest id" + self.occid)


class Owner(Hotel):  # Inheritance
    def __init(self, hotnam, onam, oid):
        super().hotelname = hotnam
        self.ownname = onam
        self.ownid = oid


print("Welcome to Hotel ABC Online Reservation ")

while True:
    choice = int(input(
        "Select one of the below options: \n 1.Add Hotel \n 2.Add Details \n 3.Add Reservation \n 4.Add Guest \n 5.Display booking : \n "))
    if (choice == 1):
        hotelname = input("Enter the hotel name:")
        a = Hotel(hotelname)
    if (choice == 2):
        name = input("enter your name:")
        ide = input("Enter your id:")
        ssn = input("Enter your ssn:")
        emp1 = Employee(name, ide, ssn)

    if (choice == 3):
        roomtype = input("What kind of room do you wantLuxurt/Normal:")
        room1 = Room("6587875", roomtype)

    if (choice == 4):
        customername = input("enter the name of Guest:")
        customerphone = input("enter the phone number of guest:")
        custid = input("Enter id number of guest")
        guest1 = Occupants(customername, customerphone, custid)

    if (choice == 6):
        break

    if (choice == 5):
        emp1.getEmployeeDetails()
        room1.GetRoomDetails()
        guest1.GetOccupantDetails()
        break



