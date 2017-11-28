import { IUserModel, IUser } from "./interface";
import { UserSchema } from "./schema";
import { Model } from "mongoose";
import { suite, test } from "mocha-typescript";
import { should } from 'chai';
import { MongoConnection } from '../../dataconnection/mongodb'


@suite
class UserTest {

    //store test data
    private data: IUser;

    //the User model
    public static User: Model<IUserModel>;

    public static before() {
        UserTest.User = new MongoConnection('User', UserSchema).model;

        //require chai and use should() assertions
        should();
    }

    constructor() {
        this.data = {
            email: "foo@bar.com",
            firstName: "Brian",
            lastName: "Love"
        };
    }

    @test("should create a new User")
    public create() {
        //create user and return promise
        return new UserTest.User(this.data).save().then(result => {
            //verify _id property exists
            result._id.should.exist;

            //verify email
            result.email.should.equal(this.data.email);

            //verify firstName
            result.firstName.should.equal(this.data.firstName);

            //verify lastName
            result.lastName.should.equal(this.data.lastName);
        });
    }

    @test("should remove a User")
    public remove() {
        //create user and return promise
        return UserTest.User.find({ email: "foo@bar.com" }).remove().then(result => {
            //result._id.should.notexist;
            //console.log(result);
            result.result.ok.should.equal(1);
        })
    }


}
