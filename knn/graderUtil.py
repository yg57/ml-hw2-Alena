# Library to do grading of Python programs.
# Percy Liang
#
# Usage:
#   grader = Grader("Name of assignment")
#   grader.addPart(name, gradeFunc, maxPoints, maxSeconds)
#   grader.grade()

import datetime, pprint, traceback, sys, signal, os
# import ipdb

defaultMaxSeconds = 10  # 10 second
TOLERANCE = 1e-4  # For measuring whether two floats are equal

# When reporting stack traces as feedback, ignore parts specific to the grading
# system.

def setTolerance(new_tol):
    TOLERANCE = new_tol

def isTracebackItemGrader(item):
    # ipdb.set_trace()
    return item[0].endswith('graderUtil.py')

def isCollection(x):
    return isinstance(x, list) or isinstance(x, tuple)

def dumpYamlOrPprint(haveYaml, x, out, yaml=None):
    if haveYaml:
        yaml.dump(x, out)
    else:
        pprint.pprint(x, stream=out)

# Return whether two answers are equal.
def isEqual(trueAnswer, predAnswer):
    # Handle floats specially
    if isinstance(trueAnswer, float) and isinstance(predAnswer, float):
        return abs(trueAnswer - predAnswer) < TOLERANCE
    # Recurse on collections to deal with floats inside them
    if isCollection(trueAnswer) and isCollection(predAnswer) and len(trueAnswer) == len(predAnswer):
        for a, b in zip(trueAnswer, predAnswer):
            if not isEqual(a, b): return False
        return True
    if isinstance(trueAnswer, dict) and isinstance(predAnswer, dict):
        if len(trueAnswer) != len(predAnswer): return False
        for k, v in trueAnswer.items():
            if not isEqual(predAnswer.get(k), v): return False
        return True

    # Numpy array comparison
    if type(trueAnswer).__name__=='ndarray':
        import numpy as np
        if isinstance(trueAnswer, np.ndarray) and isinstance(predAnswer, np.ndarray):
            if trueAnswer.shape != predAnswer.shape:
                return False
            for a, b in zip(trueAnswer, predAnswer):
                if not isEqual(a, b): return False
            return True

    # Do normal comparison
    return trueAnswer == predAnswer

# Run a function, timing out after maxSeconds.
# class TimeoutFunctionException(Exception): pass
# class TimeoutFunction:
#     def __init__(self, function, maxSeconds):
#         self.maxSeconds = maxSeconds
#         self.function = function

#     def handle_maxSeconds(self, signum, frame):
#         raise TimeoutFunctionException()

#     def __call__(self, *args):
#         old = signal.signal(signal.SIGALRM, self.handle_maxSeconds)
#         signal.alarm(self.maxSeconds + 1)
#         result = self.function(*args)
#         signal.alarm(0)
#         return result

class Part:
    def __init__(self, name, gradeFunc, maxPoints, maxSeconds):
        if not isinstance(name, str): raise Exception("Invalid name: %s" % name)
        if gradeFunc != None and not callable(gradeFunc): raise Exception("Invalid gradeFunc: %s" % gradeFunc)
        if not isinstance(maxPoints, int): raise Exception("Invalid maxPoints: %s" % maxPoints)
        if maxSeconds != None and not isinstance(maxSeconds, int): raise Exception("Invalid maxSeconds: %s" % maxSeconds)
        # Specification
        self.name = name
        self.gradeFunc = gradeFunc  # Function to call to do grading
        self.maxPoints = maxPoints  # Maximum number of points attainable on this part
        self.maxSeconds = maxSeconds  # Maximum allowed time that the student's code can take (in seconds)
        self.basic = False  # Can be changed
        # Statistics
        self.points = 0
        self.seconds = 0
        self.messages = []
        self.failed = False

    def fail(self):
        self.failed = True

class Grader:
    def __init__(self, args=sys.argv):
        self.parts = []  # Parts (to be added)
        self.manualParts = []  # Parts (to be added)
        if len(args) < 2:
            self.mode = 'all'
        else:
            self.mode = args[1]  # Either 'basic' or 'all'

        self.messages = []  # General messages
        self.currentPart = None  # Which part we're grading
        self.fatalError = False  # Set this if we should just stop immediately

    def addBasicPart(self, name, gradeFunc, maxPoints=1, maxSeconds=defaultMaxSeconds):
        part = Part(name, gradeFunc, maxPoints, maxSeconds)
        part.basic = True
        self.parts.append(part)

    def addPart(self, name, gradeFunc, maxPoints=1, maxSeconds=defaultMaxSeconds):
        if name in [part.name for part in self.parts]:
            raise Exception("Part name %s already exists" % name)
        part = Part(name, gradeFunc, maxPoints, maxSeconds)
        self.parts.append(part)

    def addManualPart(self, name, maxPoints):
        part = Part(name, None, maxPoints, None)
        self.manualParts.append(part)

    # Try to load the module (submission from student).
    def load(self, moduleName):
        try:
            return __import__(moduleName)
        except Exception as e:
            self.fail("Threw exception when importing '%s': %s" % (moduleName, e))
            self.fatalError = True
            return None
        except:
            self.fail("Threw exception when importing '%s'" % moduleName)
            self.fatalError = True
            return None

    def grade(self):
        print('========== START GRADING')
        if self.mode == 'all':
            parts = self.parts
        else:
            parts = [part for part in self.parts if part.basic]
        for part in parts:
            if self.fatalError: continue

            print('----- START PART %s' % part.name)
            self.currentPart = part

            startTime = datetime.datetime.now()
            try:
                part.gradeFunc()
            except Exception as e:
                self.fail('Exception thrown: %s -- %s' % (str(type(e)), str(e)))
                self.printException()
            
            endTime = datetime.datetime.now()
            part.seconds = (endTime - startTime).seconds
            print('----- END PART %s [took %s, %s/%s points]' % (part.name, endTime - startTime, part.points, part.maxPoints))

        totalPoints = sum(part.points for part in parts)
        maxTotalPoints = sum(part.maxPoints for part in parts)
        print('========== END GRADING [%d/%d points]' % (totalPoints, maxTotalPoints))

        try:
            import yaml
            haveYaml = True
        except ImportError:
            yaml = None
            haveYaml = False

        try:
            import dateutil.parser
            haveDateutil = True
        except ImportError:
            haveDateutil = False

        # Compute late days
        lateDays = None
        if haveYaml and haveDateutil and os.path.exists('metadata') and os.path.exists('submit.conf'):
            timestamp = datetime.datetime.fromtimestamp(os.path.getctime('metadata'))
            info = yaml.load(open('submit.conf'))
            dueDates = [assign['dueDate'] for assign in info['assignments']]
            dueDate = dateutil.parser.parse(dueDates[0])
            if timestamp > dueDate:
                lateDays = (timestamp - dueDate).days
            else:
                lateDays = 0

        result = {}
        result['mode'] = self.mode
        result['totalPoints'] = totalPoints
        result['maxTotalPoints'] = maxTotalPoints
        result['messages'] = self.messages
        if lateDays is not None:
            result['lateDays'] = lateDays
        resultParts = []
        for part in parts:
            r = {}
            r['name'] = part.name
            r['points'] = part.points
            r['maxPoints'] = part.maxPoints
            r['seconds'] = part.seconds
            r['maxSeconds'] = part.maxSeconds
            r['messages'] = part.messages
            resultParts.append(r)
        result['parts'] = resultParts
        out = open('grader-auto-%s.out' % self.mode, 'w')
        dumpYamlOrPprint(haveYaml, result, out, yaml=yaml)
        out.close()

        # Only create if doesn't exist (be careful not to overwrite the manual!)
        if len(self.manualParts) > 0:
            if not os.path.exists('grader-manual.out'):
                print("Writing %d manual parts to 'grader-manual.out'" % len(self.manualParts))
                result = {}
                resultParts = []
                for part in self.manualParts:
                    r = {}
                    r['name'] = part.name
                    r['points'] = '?'
                    r['maxPoints'] = part.maxPoints
                    r['messages'] = ['?']
                    resultParts.append(r)
                result['parts'] = resultParts
                out = open('grader-manual.out', 'w')
                dumpYamlOrPprint(haveYaml, result, out, yaml=yaml)
                out.close()
            else:
                print('grader-manual.out already exists')
        print("Total max points: %d" % (maxTotalPoints + sum(part.maxPoints for part in self.manualParts)))

    # Called by the grader to modify state of the current part

    def assignFullCredit(self):
        if not self.currentPart.failed:
            self.currentPart.points = self.currentPart.maxPoints
        return True

    def requireIsValidPdf(self, path):
        if not os.path.exists(path):
            return self.fail("File '%s' does not exist" % path)
        if os.path.getsize(path) == 0:
            return self.fail("File '%s' is empty" % path)
        fileType = os.popen('file %s' % path).read()
        if 'PDF document' not in fileType:
            return self.fail("File '%s' is not a PDF file: %s" % (path, fileType))
        return self.assignFullCredit()

    def requireIsNumeric(self, answer):
        if isinstance(answer, int) or isinstance(answer, float):
            return self.assignFullCredit()
        else:
            return self.fail("Expected either int or float, but got '%s'" % answer)

    def requireIsOneOf(self, trueAnswers, predAnswer):
        if predAnswer in trueAnswers:
            return self.assignFullCredit()
        else:
            return self.fail("Expected one of %s, but got '%s'" % (trueAnswers, predAnswer))

    def requireIsEqual(self, trueAnswer, predAnswer):
        if isEqual(trueAnswer, predAnswer):
            return self.assignFullCredit()
        else:
            return self.fail("Expected '%s', but got '%s'" % (str(trueAnswer), str(predAnswer)))

    def requireIsLessThan(self, lessThanQuantity, predAnswer ):
        if predAnswer < lessThanQuantity:
            return self.assignFullCredit()
        else:
            return self.fail("Expected to be < %f, but got %f" % (lessThanQuantity, predAnswer) )

    def requireIsTrue(self, predAnswer ):
        if predAnswer:
            return self.assignFullCredit()
        else:
            return self.fail("Expected to be true, but got false" )

    def fail(self, message):
        self.addMessage(message)
        if self.currentPart:
            self.currentPart.points = 0
            self.currentPart.fail()
        return False

    def printException(self):
        tb = [item for item in traceback.extract_tb(sys.exc_info()[2]) if not isTracebackItemGrader(item)]
        for item in traceback.format_list(tb):
            self.fail('%s' % item)

    def addMessage(self, message):
        print(message)
        if self.currentPart:
            self.currentPart.messages.append(message)
        else:
            self.messages.append(message)
