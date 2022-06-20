const express = require('express')
const app = express()

const fs = require('fs')

var multer = require('multer')
var upload = multer({ dest: 'new_dataset/audio'})

var ipfsAPI = require('ipfs-api');
const path = require('path')
// const bodyparser = require('body-parser')
var ipfs = ipfsAPI('127.0.0.1', '5001', {protocol: 'http'});
// app.use(bodyparser.urlencoded({ extended : false}));
app.use(express.static('public'));
app.use('/js', express.static(__dirname + "/js"));

app.get('/', function (req, res) {
    res.sendFile(__dirname + '/public/index.html')
});

app.post('/docs', function (req, res) {
    res.sendFile(__dirname + '/public/smartEwillService.html')
});

app.post('/image', upload.single('image'), function (req, res, next) {
    // console.log(req.file);
    var data = Buffer.from(fs.readFileSync(req.file.path))
    ipfs.add(data, function (err, file) {
        if(err) {
            console.log(err)
        }
        // console.log('file', file)
        res.send(
            '<!DOCTYPE html>'
            + '<html lang="en" class="h-100">'
            + '<head>'
            + '<meta charset="utf-8">'
            + '<meta name="viewport" content="width=device-width, initial-scale=1">'
            + '<meta http-equiv="X-UA-Compatible" content="ie=edge">'
            + '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuzKMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">'
            + '<title>Notarization Service</title>'
            + '</head>'
            + '<body class="p-3 mb-2 bg-dark text-white">'
            + '<h1>유서공증 서비스</h1>'
            + '<p> URL: https://ipfs.io/ipfs/' + file[0].hash + '</p>'
            + '<button type="submit" class="btn btn-primary">NFT minting</button>'
            + '</body>'
            + '</html>'
        );
    });
});

app.post('/voice', upload.single('voice'), function (req, res, next) {
    console.log(req.file);
    var data = Buffer.from(fs.readFileSync(req.file.path))
    ipfs.add(data, function (err, file) {
        if(err) {
            console.log(err)
        }
        console.log(file)
        res.send(
            '<!DOCTYPE html>'
            + '<html lang="en" class="h-100">'
            + '<head>'
            + '<meta charset="utf-8">'
            + '<meta name="viewport" content="width=device-width, initial-scale=1">'
            + '<meta http-equiv="X-UA-Compatible" content="ie=edge">'
            + '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuzKMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">'
            + '<title>Notarization Service</title>'
            + '</head>'
            + '<body class="p-3 mb-2 bg-dark text-white">'
            + '<h1>유서공증 서비스</h1>'
            + '<p> URL: https://ipfs.io/ipfs/' + file[0].hash + '</p>'
            + '<button type="submit" class="btn btn-primary">NFT minting</button>'
            + '</body>'
            + '</html>'
        );
    });
});


app.listen(3000, () => console.log('3000 번으로 연결됨'))
    
