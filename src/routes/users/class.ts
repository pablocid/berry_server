export class User {
  public _id: string;
  public email: string;
  public firstName: string;
  public lastName: string;
  public created: Date;
  public password: string;
  public role: string;

  constructor(obj?: any) {
    if (obj) {
      if (obj._id) { this._id = obj.id }
      if (obj.email) { this.email = obj.email }
      if (obj.firstName) { this.firstName = obj.firstName }
      if (obj.lastName) { this.lastName = obj.lastName }
      if (obj.password) { this.password = obj.password }
      if (obj.role) { this.role = obj.role }
    }
  }

  public payload() {
    return {
      _id: this._id,
      role: this.role
    }
  }

  public json() {
    let obj: any = {}
    if (this._id) { obj._id = this._id; }
    obj.email = this.email;
    obj.firstName = this.firstName;
    obj.lastName = this.lastName;
    obj.role = this.role;
    obj.password = '***';

    return obj;
  }

}