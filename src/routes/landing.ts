import { NextFunction, Request, Response, Router } from "express";
import { BaseRoute } from "../models/class.route";
import * as express from "express";
import * as Multer from 'multer';
import { DataPartial } from '../models/DataPartial';
import * as Q from 'q';

/**
 * / route
 *
 * @class LandingRoute
 */
export class LandingRoute extends BaseRoute {

    public static multer(value: string) {
        return Multer({ dest: 'uploads/' }).single(value)
    }

    /**
     * Create the routes.
     *
     * @class LandingRoute
     * @method create
     * @static
     */
    public static create(router: Router) {

        //add home page route
        router.get("/", (req: Request, res: Response, next: NextFunction) => {
            new LandingRoute().index(req, res, next);
        });
    }

    public static get route() {
        let r = express.Router();
        r.get("/", (req: Request, res: Response, next: NextFunction) => {
            new LandingRoute().index(req, res, next);
        });

        r.post("/form_response", LandingRoute.multer('archivo'), (req: Request, res: Response, next: NextFunction) => {
            new LandingRoute().formResponse(req, res, next);
        });
        return r;
    }

    /**
     * Constructor
     *
     * @class LandingRoute
     * @constructor
     */
    constructor() {
        super();
    }

    /**
     * The home page route.
     *
     * @class LandingRoute
     * @method index
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public index(req: Request, res: Response, next: NextFunction) {
        //set custom title
        this.title = "Home | MyExpress";
        var self = this;
        (async () => {
            try {
                let model:any = {};
               model = await self._dataPartials();

                model['message'] = 'Welcome to my App';
                model['pepe'] = 'Pepedfh';

                self.render(req, res, "pages/index", model);
            } catch (e) {
                console.error(e.message);
                res.json(e);
            }
        })();
        //ASYNC();
    }

    private _dataPartials():Q.Promise<any>{

        var dataPartial: DataPartial[] = [
            new DataPartial('footerTopArea'),
            new DataPartial('topBar'),
            new DataPartial('menuBar')
        ]
        return Q.all(dataPartial.map(x=>x.dataTemplate)).then((x)=>{
            var obj:any = {};
            for (var i = 0; i < x.length; i++) { obj[dataPartial[i].id]= x[i]; }
            return obj;
        });

    }

    public formResponse(req: Request, res: Response, next: NextFunction) {
        res.json({
            path: req.file.path,
            originalName: req.file.originalname,
            encoding: req.file.encoding,
            minetype: req.file.mimetype,
            size: req.file.size,
            destination: req.file.destination,
            filename: req.file.filename,
            fieldname: req.file.fieldname,
            buffer: req.file.buffer
        });
    }
}