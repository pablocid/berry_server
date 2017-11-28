var fs = require('fs');
import { DataPartialRoute } from '../routes/partials';
import * as Q from 'q';

export class DataPartial {
    public id: string;
    public filename: string;

    private _pr: DataPartialRoute;
    constructor(id: string) {
        this.id = id;
        this.filename = this.toDashCase(id);
        this._pr = new DataPartialRoute();
    }

    public get data() {
        return this._pr.getJSONByName(`${this.filename}.json`);
    }
    public get schema() {
        return this._pr.getJSONByName(`${this.filename}.schm.json`);
    }

    public get dataTemplate() {
        return Q.spread([this.data, this.schema], (data, schema) => {
            return {
                data,
                schema,
                postUrl: `api/partials/${this.filename}`,
                id: this.id
            }
        });
    }

    private toDashCase(value: string): string {
        return value.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
    }
}
