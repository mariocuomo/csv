function showIntroduzione() {
    var rp = document.getElementById("introduzione");
    var sh = document.getElementById("introduzione_hide");
  
    if (rp.style.display === "none") {
        rp.style.display = "block";
        sh.textContent='[-]';
    } else {
        rp.style.display = "none";
        sh.textContent='[+]';
    }
  }

function showcasoduso() {
var rp = document.getElementById("casoduso");
var sh = document.getElementById("casoduso_hide");

if (rp.style.display === "none") {
    rp.style.display = "block";
    sh.textContent='[-]';
} else {
    rp.style.display = "none";
    sh.textContent='[+]';
}
}

function showUserBased() {
    var rp = document.getElementById("userBased");
    var sh = document.getElementById("userBased_hide");
    
    if (rp.style.display === "none") {
        rp.style.display = "block";
        sh.textContent='[-]';
    } else {
        rp.style.display = "none";
        sh.textContent='[+]';
    }
}

function showItemBased() {
    var rp = document.getElementById("itemBased");
    var sh = document.getElementById("itemBased_hide");
    
    if (rp.style.display === "none") {
        rp.style.display = "block";
        sh.textContent='[-]';
    } else {
        rp.style.display = "none";
        sh.textContent='[+]';
    }
}

function showPerTe() {
    var rp = document.getElementById("perte");
    var sh = document.getElementById("perte_hide");
    
    if (rp.style.display === "none") {
        rp.style.display = "block";
        sh.textContent='[-]';
    } else {
        rp.style.display = "none";
        sh.textContent='[+]';
    }
}

function showNoMean() {
    var rp = document.getElementById("noMean");
    var sh = document.getElementById("noMean_hide");
    
    if (rp.style.display === "none") {
        rp.style.display = "block";
        sh.textContent='[-]';
    } else {
        rp.style.display = "none";
        sh.textContent='[+]';
    }
}

