from datetime import datetime, timezone, timedelta;

class Curtimeout():
    def __init__(self, timezone=timezone.utc):
        """_"time based float or str generator"_

        Args:
            timezone (_timezone_, optional): _timezone_. Defaults to timezone.utc.
        """
        inittime=datetime.now(timezone);
        self.float=self.__tonum(inittime);
        self.str=self.__tostr(inittime);
    def __tonum(self, datetime):
        return float(datetime.strftime("%Y%m%d%H%M%S.%f"));
    def __tostr(self, datetime):
        return datetime.strftime("%Y-%m-%d_%H-%M-%S_%f");

Starttime=Curtimeout();

class Tpass(Curtimeout):
    def __init__(self, TimeAtRun):
        """_"Time difference between start and calltime"_

        Args:
            TimeAtRun (_Curtimeout.float_): _current time value_
        """
        Curtimeout.__init__(self);
        try:
            self.tdiff=Starttime.float-TimeAtRun.float;
        except:
            print("Must set Starttime class first.");
            raise ValueError;
        timeres=self.__calcpart(self);
        self.years=timeres[5];
        self.months=timeres[4];
        self.days=timeres[3];
        self.hrs=timeres[2];
        self.mins=timeres[1];
        self.secs=timeres[0];
        self.msecs=(self.tdiff)-round(self.tdiff);
        
    def __calcpart(self):
        tdiff=self.tdiff;
        decidegits=[2, 4]; numsoftimesap=1+1+1+1+1+1;
        timeres=[];
        for ind in range(0, numsoftimesap, 1):
            if ind<5:
                decind=0;
            else:
                decind=1;
            timeres[ind]=int((tdiff*(10**(decidegits[decind]*ind)))%decidegits[decind]);
        return timeres;