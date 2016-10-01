/**
 * Created by nate on 9/26/2016.
 */
var data = source.data;
var data_keys = Object.keys(data).sort()
var filetext = data_keys.join() + '\n';
for (i=0; i < data[data_keys[0]].length; i++) {
    var currRow = [data['Meta'][i].toString(),
                   data['Meta'][i].toString(),
                   data['Meta'][i].toString().concat('\n')];
    var joined = currRow.join();

    var joined = ''
    for (j=0; j < data_keys.length; j++) {
        joined = joined.concat(data[data_keys[j]][i].toString()).concat(',')
    }
    joined = joined.concat('\n')

    filetext = filetext.concat(joined);
}

var filename = 'download_data.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}