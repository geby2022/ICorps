var Employee = { //객체 리터럴 표기법
    //필수 정보는 근무자 이름, 주소, 연락처, 사업체명, 서명
    name : '',
    address : '',
    phone : '',
    companyName:'',
    signDataURL:'',

    //선택입력정보
    start_year: '',
    start_month: '',
    start_day: '',
    end_year:'',
    end_month:'',
    end_day:'',

    work_place:'', //write
    work_content:'', //write
    work_start_time:'', //시간 선택
    work_end_time:'', //시간 선택

    work_day_num:'', //근무일 언제임 option
    holiday:'',//휴일 언제임 option 
    pay_format:'',//임금형태 뭐임 option

    salary:'', // 월급
    bonus:'', //보너스 있음 가격은 (option에서 있음에 체크할 경우)
    etc_payCheck:'',//기타급여 있음 없음 (체크에 따라서 true false)
    pay_date:'',//임금 지급일
    
    payWay:'',//임금 지급방법 option
    _4insurance:'',//4대보험 option

    writtenYear:'', //작성년월일
    writtenMonth:'',
    writtenDay:''
    
};

function renewalDate(year,month,day){ // 현재 년월일을 가져오는 함수
    let today = new Date();
    document.getElementById(year).value = today.getFullYear(); //년
    document.getElementById(month).value = today.getMonth() + 1;//월
    document.getElementById(day).value = today.getDate(); //일
} 



function checkRadio(form) {
    for (var i=0; i<form.length; i++) {
     if (form[i].checked == true) {
            return form[i].value;;
          }
      }
      
  }

  function checkBox(form) {
    var str='';
    for (var i=0; i<form.length; i++) {
     if (form[i].checked == true) {
            str += form[i].value;;
            str += ', '
          }
      }
      str=str.substr(0, str.length -2); //콤마 지우기
      return str;
  }

  function jsonPrint(form){ //값이 있을 때만 출력하자 //미입력된 항목으로 나오게 해주는 곳
    if (form==null)
    {
        form='';
        return form;
    }

    else
    {
        return form;
    }
  }

function signReset(btn_id)
{
    var btn = document.getElementById(btn_id);
    btn.addEventListener('click',function(){
        alert("초기화");
    })
}

function downloadURI(uri, name){ //서명 저장
	var link = document.createElement("a")
	link.download = name;
	link.href = uri;
	document.body.appendChild(link);
	link.click();
}



function Signature(){
    var canvas = $("#signature")[0];
    var signature = new SignaturePad(canvas, {
        minWidth : 2,
        maxWidth : 2,
        penColor : "rgb(0, 0, 0)"
    });

    $("#reset_sign").on("click", function() {
        signature.clear();
    });
    
    $("#make_contract").on("click", function() {
        if(signature.isEmpty()) {
            Employee.signDataURL = '';
        } else {
            var data = signature.toDataURL("image/png"); //캠버스 영역을 base64값으로 즉시 변환
            Employee.signDataURL = data;  // 컨트랙트 txt에 이미지의 base64 문자열 저장.
            //downloadURI(data,"sign.png"); // 서명 파일로 저장 (사실 base64 문자열만 저장해도 무방함)

        }
    });

}
  
window.onload = function(){ //webpage의 모든 구성요소에 대한 로드가 끝났을 때 브라우저에 의해서 호출되는 함수(이벤트)
    
    renewalDate('write_year','write_month','write_day');// 작성일 갱신 함수
    Signature(); //서명 함수
    
    var contract = document.getElementById('make_contract');
    contract.addEventListener('click',function(){
        
            //필수입력 항목 
            Employee.name = document.getElementById('EmployeeModel_EmployeeName').value;
            Employee.address = document.getElementById('EmployeeModel_EmployeeAddress').value;
            Employee.phone = document.getElementById('EmployeeModel_EmployeeContactUs').value;
            Employee.companyName = document.getElementById('BossModel_CompanyName').value;

            //선택입력 항목
            /* 계약기간 */
            Employee.start_year= jsonPrint(document.getElementById('StandardModel_ContractStartDate_Value_Year').value);
            Employee.start_month=jsonPrint(document.getElementById('DevStMonth').value);
            Employee.start_day= jsonPrint(document.getElementById('DevStDay').value);


            Employee.end_year= jsonPrint(document.getElementById('StandardModel_ContractEndDate_Value_Year').value);
            Employee.end_month= jsonPrint(document.getElementById('DevEdMonth').value);
            Employee.end_day= jsonPrint(document.getElementById('DevEdDay').value);

            /*근무장소,업무내용, 근로 시작시간, 근로 종료시간 */
            Employee.work_place= jsonPrint(document.getElementById('workPlace').value);
            Employee.work_content= jsonPrint(document.getElementById('workContent').value);
            Employee.work_start_time= jsonPrint(document.getElementById('StandardModel_WorkStartTime').value);
            Employee.work_end_time= jsonPrint(document.getElementById('StandardModel_WorkEndTime').value);
            
            if(Employee.work_start_time == "" ||Employee.work_end_time == "" ) // 둘중하나라도 안적으면 미입력한 항목입니다.
            {
                Employee.work_start_time ='';
                Employee.work_end_time = '';

            }
            else //아니면 물결표시 추가
            {
                Employee.work_end_time =" ~ " + jsonPrint(document.getElementById('StandardModel_WorkEndTime').value);
            }
    
            /*근무일 및 휴일*/
            Employee.work_day_num= checkRadio(document.getElementsByName('StandardModel.WeekWorkDateCnt'));

            if(Employee.holiday = checkBox(document.getElementsByName('WeekWorkHoliday'))!='')
            {
                Employee.holiday= checkBox(document.getElementsByName('WeekWorkHoliday'));
            }
            else
            {
                Employee.holiday= '';
            
            }

    


            /*임금 형태 */
            Employee.pay_format= checkRadio(document.getElementsByName('StandardModel.PayTypeCode'));
            Employee.pay_amount= jsonPrint(document.getElementById('pay_amount').value); //임금
            if( Employee.pay_amount!='')
                {
                    Employee.pay_amount= jsonPrint(document.getElementById('pay_amount').value) + " 원 ";
                }

            /*보너스*/
           // var isBonus = checkRadio(document.getElementsByName('StandardModel.IsBonus'));
            if(checkRadio(document.getElementsByName('StandardModel.IsBonus'))=='상여금 있음') //보너스 있다고 체크 했을 때만 금액 나옴
            {
                if(jsonPrint(document.getElementById('bonus').value)<=0)
                {
                    Employee.bonus= "상여금 있음 "+jsonPrint(document.getElementById('bonus').value);
                }
                else
                {
                    Employee.bonus= "상여금 "+jsonPrint(document.getElementById('bonus').value)+"원";
                }
            }
            else
            {
                if(checkRadio(document.getElementsByName('StandardModel.IsBonus'))=='상여금 없음')
                {
                    Employee.bonus = '상여금 없음';
                }
                else
                {
                    Employee.bonus='';
                }
            }
            
            /*기타 급여*/
            if(checkRadio(document.getElementsByName('StandardModel.IsEtcPay'))=='기타 급여 있음') //기타급여 있다고 체크할 때만 있다고 나옴
            {
                Employee.etc_payCheck=checkRadio(document.getElementsByName('StandardModel.IsEtcPay'));
            }
            else if(checkRadio(document.getElementsByName('StandardModel.IsEtcPay'))=='기타 급여 없음')
            {
                Employee.etc_payCheck='기타 급여 없음';
            }
            else
            {
                Employee.etc_payCheck='';
            }

            /*임금 지급일*/
            if(checkRadio(document.getElementsByName('StandardModel.PayGiveDayType'))=='매월') //매월
            {
                Employee.pay_date= jsonPrint(document.getElementById('month_pay_date').value);
                if( Employee.pay_date!='')
                {
                    Employee.pay_date= "매월 "+ jsonPrint(document.getElementById('month_pay_date').value) + " 일 ";
                }
            }
            else if(checkRadio(document.getElementsByName('StandardModel.PayGiveDayType'))=='매주') //매주
            {
                Employee.pay_date= "매주"; //pay format은 저장만 하고 출력으로 활용하는건 pay_date뿐 
            }
            else if(checkRadio(document.getElementsByName('StandardModel.PayGiveDayType'))=='매일') //매일
            {
                Employee.pay_date= "매일";
            }
            else
            {
                Employee.pay_date= '';
            }

            /*지급 방법*/
            Employee.payWay= checkRadio(document.getElementsByName('StandardModel.PayGiveModeType'));
            
            /*4대 보험 적용여부*/
            if(Employee._4insurance= checkBox(document.getElementsByName('insurance_4'))!='')
            {
                Employee._4insurance= checkBox(document.getElementsByName('insurance_4'));
            }
            else
            {
                Employee._4insurance = '';
            }

            Employee.writtenYear= jsonPrint(document.getElementById('write_year').value);
            Employee.writtenMonth= jsonPrint(document.getElementById('write_month').value);
            Employee.writtenDay= jsonPrint(document.getElementById('write_day').value);

    
        

        if((Employee['name']&&Employee['address']&&Employee['phone']&&Employee['companyName'])!=''&&Employee['signDataURL']!='') //필수입력공간이 전부 null이 아닐 때만
        {
            var recardianData = JSON.stringify(Employee,null,3); // 제이슨 객체를 String 객체로 변환, 반대로 JSON.parse()는 string 객체를 json 객체로 변환한다. null과 3은 정렬을 위해서다.
            //saveToFile_Chrome(Employee.name+" 고객 계약서.txt",recardianData); // Chrome 환경에서, 오른쪽 인수의 내용을 txt 파일로 다운로드한다, download 폴더 경로로 저장된다.
            localStorage.setItem('employee',recardianData);
            alert('컨트랙트 생성 완료');
            location.href='CreatedContract.html'
        }
        else
        {
            localStorage.setItem('employee','');
            alert('필수 항목을 반드시 입력해주세요');
        }

    })
}
    