import { BaseRoute } from '../../models/class.route'
import { NextFunction, Request, Response, Router } from 'express';
import { FileSysConnection } from '../../dataconnection/filesys';

export class DataPartialRoute extends BaseRoute {
    /**
     * Create the routes.
     *
     * @class DataPartialRoute
     * @method route
     * @static
     */
    public static get route() {
        let r = Router();
        var obj = new DataPartialRoute();

        r.get("/", (req: Request, res: Response, next: NextFunction) => {
            obj.index(req, res, next);
        });
        r.get("/:name", (req: Request, res: Response, next: NextFunction) => {
            obj.partial(req, res, next);
        });
        r.post("/:name", (req: Request, res: Response, next: NextFunction) => {
            console.log('partial name post ')
            obj.update(req, res, next);
        });

        return r;
    }

    /**
     * Constructor
     *
     * @class DataPartialRoute
     * @constructor
     */
    private _file: FileSysConnection;
    constructor() {
        super();
        this._file = new FileSysConnection();
    }

    /**
     *
     * @class DataPartialRoute
     * @method index
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public index(req: Request, res: Response, next: NextFunction) {
        //lista de los routes disponibles
        res.json({ lista: [1, 2, 3, 4, 5] });
    }

    /**
     *
     * @class DataPartialRoute
     * @method partial
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public partial(req: Request, res: Response, next: NextFunction) {
        let name = req.params.name;
        this.getJSONByName(name)
            .then((data) => {
                res.json(data);
            })
            .catch((err) => {
                res.json(err);
            });
    }

    public update(req: Request, res: Response, next: NextFunction) {
        //lista de los routes disponibles
        let data = req.body.data;
        let name = req.params.name;
        if (!data || !name) { res.status(404).json({ err: "no data" }); return; }
        this.setJSONByName(name, data)
            .then((anw) => {
                res.status(200).json(anw);
            })
            .catch((err) => {
                res.status(404).json(err);
            })
    }

    public getJSONByName(name: string) {
        return this._file.getJSON(name);
    }

    public setJSONByName(name: string, data: JSON) {
        return this._file.setJSON(name, data);
    }

}