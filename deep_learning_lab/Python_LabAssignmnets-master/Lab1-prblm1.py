

def Check(passwd):
   
    symbols=['$','@','#','%','!']
    flag=True
    if len(passwd) < 6:
        print('the length of password must be at least 6 chars long')
        flag=False
    if len(passwd) >16:
        print('the length of password must be not be greater than 8')
        flag=False
    if not any(char.isdigit() for char in passwd):
        print('Password should contain at least one number')
        flag=False
    if not any(char.isupper() for char in passwd):
        print('Password should contain at least one uppercase letter')
        flag=False
    if not any(char.islower() for char in passwd):
        print('Password should contain at least one lowercase letter')
        flag=False
    if not any(char in symbols for char in passwd):
        print('the password should have at least one special character')
        flag=False
    if return_val:
        print('Ok')
    return flag
username = input('Enter Username : ')
passwd = input('Enter Password : ')
print(Check(passwd))












