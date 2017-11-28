import { createConnection } from 'mysql';
import { defer as Defer } from 'q';

export class MySQLConnection {
  private _connection: any;

  constructor() {
    this._connection = createConnection({
      host: 'localhost',
      database:'myexpress',
      user: 'root',
      password: 'cid123'
    });
  }

  getUsers() {

    this._connection.connect();

    var DF = Defer();


    this._connection.query('SELECT * FROM users',  (err: any, rows: any[], fields: string[]) => {
      if (err) {
        this._connection.end();
        DF.reject(err);
      }

      console.log('The solution is: ', rows[0].firstName);
      this._connection.end();
      DF.resolve(rows);
    });

    return DF.promise;
  }

}
